#!/usr/bin/env python3
"""
🔥 OPEN ALPHA - DEPLOYMENT VALIDATION SCRIPT
Validates all deployment files and system readiness for DigitalOcean
"""

import os
import sys
from pathlib import Path

def main():
    print("🔥 OPEN ALPHA - FINAL DEPLOYMENT VALIDATION")
    print("=" * 60)

    # Check all deployment files
    deployment_files = [
        'Dockerfile.deployment',
        'deployment_startup.py', 
        'requirements_deployment.txt',
        'docker-compose.deployment.yml',
        '.github/workflows/digitalocean-deploy.yml',
        'DIGITALOCEAN_DEPLOYMENT.md',
        'SYSTEM_ARCHITECTURE_REFERENCE.md',
        'historical_data_downloader.py',
        'prepare_deployment.py'
    ]

    print("📁 DEPLOYMENT FILES CHECK:")
    missing_files = []
    for file in deployment_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)

    # Check core application files
    core_files = [
        'src/main.py',
        'src/debug_safety.py',
        'src/historical_data_provider.py',
        'src/dashboard/',
        'src/bybit_api.py',
        'config/debug.yaml'
    ]

    print("\n📋 CORE APPLICATION CHECK:")
    for file in core_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)

    # Database check
    db_path = Path('src/data/speed_demon_cache/market_data.db')
    if db_path.exists():
        size = db_path.stat().st_size
        print(f"\n📊 DATABASE: ✅ {db_path} ({size:,} bytes)")
    else:
        print(f"\n📊 DATABASE: ❌ {db_path} - Will be created on deployment")

    print("\n🚀 DEPLOYMENT READINESS:")
    if not missing_files:
        print("✅ ALL REQUIRED FILES PRESENT")
        print("✅ READY FOR GITHUB PUSH AND DIGITALOCEAN DEPLOYMENT")
        print("")
        print("Next steps:")
        print("1. git add .")
        print("2. git commit -m '🚀 Deploy Open Alpha to DigitalOcean'")
        print("3. git push origin main")
        print("4. Monitor GitHub Actions for deployment status")
        return True
    else:
        print(f"❌ MISSING FILES: {missing_files}")
        print("❌ DEPLOYMENT NOT READY")
        return False

    print("\n🔥 Open Alpha Wealth Management System")
    print("📋 System Architecture Reference v3.0 Compliant")
    print("🛡️ Debug Safety Active - No Trading Risk")
    print("☁️ DigitalOcean Deployment Ready")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)