#!/usr/bin/env python3
"""
DigitalOcean Deployment URL Finder
Helps locate the correct URL for your deployed application
"""

import requests
import concurrent.futures
import time
from urllib.parse import urljoin

# Common DigitalOcean App Platform URL patterns
url_patterns = [
    # Standard patterns
    "https://bybit-bot-{hash}.ondigitalocean.app",
    "https://bybit-bot-fresh-{hash}.ondigitalocean.app", 
    "https://bybit-bot-{random}.ondigitalocean.app",
    
    # With repository name
    "https://zvxndr-bybit-bot-{hash}.ondigitalocean.app",
    "https://bybit-bot-zvxndr-{hash}.ondigitalocean.app",
    
    # Short hash patterns (common)
    "https://bybit-bot-{short}.ondigitalocean.app",
    "https://bybit-{hash}.ondigitalocean.app",
    
    # Branch-based
    "https://bybit-bot-main-{hash}.ondigitalocean.app",
    "https://main-bybit-bot-{hash}.ondigitalocean.app"
]

# Generate hash variations from recent commit
commit_hash = "f9b7baf"  # From logs
hash_variations = [
    commit_hash[:6],   # f9b7ba
    commit_hash[:7],   # f9b7baf  
    commit_hash[:8],   # f9b7baf5
    commit_hash,       # Full hash
    "latest",
    "main",
    "production",
    "app"
]

def check_url(url):
    """Check if URL is accessible and return status"""
    try:
        # Try health endpoint first
        health_url = urljoin(url, '/health')
        response = requests.get(health_url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return f"✅ FOUND: {url} (Health: OK)"
        
        # Try root endpoint
        response = requests.get(url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return f"✅ FOUND: {url} (Status: {response.status_code})"
        elif response.status_code in [301, 302, 307, 308]:
            return f"🔄 REDIRECT: {url} -> {response.headers.get('Location', 'Unknown')}"
        else:
            return f"❌ {url} (Status: {response.status_code})"
            
    except requests.exceptions.RequestException as e:
        return f"❌ {url} (Error: {str(e)[:50]})"

def main():
    print("🔍 Searching for your DigitalOcean deployment URL...")
    print(f"📋 Using commit hash: {commit_hash}")
    print("=" * 60)
    
    # Generate all possible URLs
    urls_to_check = []
    
    for pattern in url_patterns:
        for hash_var in hash_variations:
            url = pattern.format(hash=hash_var, short=hash_var[:6], random=hash_var)
            urls_to_check.append(url)
    
    # Remove duplicates
    urls_to_check = list(set(urls_to_check))
    
    print(f"🚀 Checking {len(urls_to_check)} possible URLs...")
    print("⏱️  This may take 30-60 seconds...")
    print()
    
    # Check URLs in parallel
    successful_urls = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(check_url, url): url for url in urls_to_check}
        
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            print(result)
            
            if "✅ FOUND" in result:
                successful_urls.append(result.split()[2])  # Extract URL
    
    print("\n" + "=" * 60)
    
    if successful_urls:
        print("🎉 SUCCESS! Found working URLs:")
        for url in successful_urls:
            print(f"   🌐 {url}")
            print(f"   🏥 Health Check: {url}/health")
            print(f"   📊 Dashboard: {url}/")
    else:
        print("❌ No accessible URLs found.")
        print("\n💡 Next steps:")
        print("   1. Check DigitalOcean App Platform console")
        print("   2. Look for 'Live App' or 'View' button")
        print("   3. Custom domains may be configured")
        print("   4. App might still be deploying (wait 2-3 minutes)")

if __name__ == "__main__":
    main()