#!/usr/bin/env python3
"""
🌐 ENHANCED DEPLOYMENT URL FINDER
================================

Comprehensive checker for DigitalOcean App Platform URLs.
Tries many common patterns and provides deployment guidance.
"""

import requests
import time
from datetime import datetime

def check_deployment_urls():
    """Check comprehensive list of possible DigitalOcean URLs"""
    
    # Repository info
    repo_name = "bybit-bot"
    
    # Comprehensive list of URL patterns
    url_patterns = [
        # Basic patterns
        f"https://{repo_name}.ondigitalocean.app",
        f"https://{repo_name}-app.ondigitalocean.app",
        f"https://{repo_name}-fresh.ondigitalocean.app",
        
        # With hyphens replaced
        "https://bybitbot.ondigitalocean.app",
        "https://bybitbot-app.ondigitalocean.app", 
        "https://bybitbot-fresh.ondigitalocean.app",
        
        # Common variations
        "https://bybit.ondigitalocean.app",
        "https://trading-bot.ondigitalocean.app",
        "https://ai-trading-bot.ondigitalocean.app",
        
        # With numbers/versions
        "https://bybit-bot-v1.ondigitalocean.app",
        "https://bybit-bot-v2.ondigitalocean.app",
        "https://bybit-bot-main.ondigitalocean.app",
        
        # Alternative naming
        "https://bybit-trading.ondigitalocean.app",
        "https://bybit-api.ondigitalocean.app",
        "https://bybit-dashboard.ondigitalocean.app",
        
        # Random hash patterns (DO sometimes uses)
        "https://bybit-bot-12345.ondigitalocean.app",
        "https://app-12345.ondigitalocean.app",
        
        # Custom subdomains
        "https://zvxndr-bybit-bot.ondigitalocean.app",
        "https://willi-bybit-bot.ondigitalocean.app",
    ]
    
    print("🌐 COMPREHENSIVE DIGITALOCEAN URL SEARCH")
    print("=" * 55)
    print(f"🔍 Checking {len(url_patterns)} possible URL patterns...")
    print(f"⏰ Time: {datetime.now().isoformat()}")
    print()
    
    working_urls = []
    partially_working = []
    
    for i, url in enumerate(url_patterns, 1):
        print(f"[{i:2d}/{len(url_patterns)}] 🌐 {url}")
        
        try:
            # Quick health check first
            response = requests.get(f"{url}/health", timeout=8)
            if response.status_code == 200:
                print(f"         ✅ HEALTHY! Status 200 - Health endpoint working")
                working_urls.append((url, "healthy"))
                continue
            elif 200 <= response.status_code < 300:
                print(f"         🟢 WORKING! Status {response.status_code}")
                working_urls.append((url, "working"))
                continue
            elif 400 <= response.status_code < 500:
                print(f"         🟡 APP EXISTS! Status {response.status_code} (health endpoint missing)")
                partially_working.append((url, f"status_{response.status_code}"))
                
                # Try root endpoint
                try:
                    root_response = requests.get(url, timeout=5)
                    if 200 <= root_response.status_code < 300:
                        print(f"         ✅ ROOT WORKS! Status {root_response.status_code}")
                        working_urls.append((url, "root_working"))
                        continue
                except:
                    pass
            else:
                print(f"         ❌ Status {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"         ⏰ Timeout - May be starting up")
            
        except requests.exceptions.ConnectionError:
            print(f"         🔌 No connection")
            
        except Exception as e:
            print(f"         ❓ Error: {str(e)[:50]}")
    
    # Generate report
    print("\n" + "=" * 55)
    print("📊 DEPLOYMENT URL SEARCH RESULTS")
    print("=" * 55)
    
    if working_urls:
        print("🎉 FOUND WORKING APPS:")
        for url, status in working_urls:
            status_icon = "✅" if status == "healthy" else "🟢" 
            print(f"   {status_icon} {url} ({status})")
        print()
        print("🎯 USE THESE URLS TO ACCESS YOUR APP!")
        
    elif partially_working:
        print("🟡 FOUND APPS (may be starting):")
        for url, status in partially_working:
            print(f"   🟡 {url} ({status})")
        print()
        print("💡 Try these URLs - the app might still be starting up!")
        
    else:
        print("❌ NO DEPLOYED APPS FOUND")
        print()
        print("🔧 TROUBLESHOOTING STEPS:")
        print("   1. Check DigitalOcean App Platform console")
        print("   2. Verify the app name in your DO dashboard")
        print("   3. Check if deployment is still in progress")
        print("   4. Look for custom domain settings")
        print("   5. Verify GitHub repository is connected")
        print()
        print("🔍 If you have the DO console open, look for:")
        print("   - App name (might be different from repo name)")
        print("   - Custom domains configured")
        print("   - Deployment status and logs")
    
    return working_urls or partially_working

def main():
    results = check_deployment_urls()
    
    if results:
        print("\n🚀 NEXT STEPS:")
        print("   1. Visit the working URL(s) above")
        print("   2. Test the unified dashboard interface") 
        print("   3. Verify chart styling and configuration save")
        print("   4. Check that testnet balance displays correctly")
    else:
        print("\n⏳ If deployment is in progress, wait 5-10 minutes and try again.")
        print("   Deployment logs should show the container building successfully.")

if __name__ == "__main__":
    main()