#!/usr/bin/env python3
"""
🔗 DEPLOYMENT URL CHECKER
========================

Helps identify your actual DigitalOcean App Platform URL.
Common patterns for DigitalOcean apps.
"""

import requests
import time

def check_common_urls(project_name="bybit-bot"):
    """Check common DigitalOcean URL patterns"""
    
    # Common DigitalOcean URL patterns
    possible_urls = [
        f"https://{project_name}.ondigitalocean.app",
        f"https://{project_name}-app.ondigitalocean.app", 
        f"https://{project_name}-fresh.ondigitalocean.app",
        f"https://bybit-bot.ondigitalocean.app",
        f"https://bybit-bot-app.ondigitalocean.app",
        f"https://bybit-bot-fresh.ondigitalocean.app",
        # Add more patterns as needed
    ]
    
    print("🔗 CHECKING COMMON DIGITALOCEAN URLS")
    print("=" * 50)
    
    working_urls = []
    
    for url in possible_urls:
        print(f"🌐 Trying: {url}")
        
        try:
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                print(f"   ✅ FOUND! Status 200")
                working_urls.append(url)
            elif 400 <= response.status_code < 500:
                print(f"   ⚠️ App exists but health endpoint not found (status {response.status_code})")
                # Still might be the right URL, just no health endpoint
                working_urls.append(url)
            else:
                print(f"   ❌ Status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"   🔌 No connection")
        except requests.exceptions.Timeout:
            print(f"   ⏰ Timeout")
        except Exception as e:
            print(f"   ❓ Error: {e}")
            
    print("\n" + "=" * 50)
    
    if working_urls:
        print("🎉 FOUND WORKING URLS:")
        for url in working_urls:
            print(f"   ✅ {url}")
        print("\nUpdate your deployment_monitor.py with the correct URL!")
    else:
        print("❌ NO WORKING URLS FOUND")
        print("\nPossible reasons:")
        print("1. App is still building/deploying")
        print("2. Different app name in DigitalOcean")
        print("3. Custom domain configured")
        print("\nCheck your DigitalOcean App Platform console for the actual URL.")

def main():
    check_common_urls()

if __name__ == "__main__":
    main()