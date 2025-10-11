# 🔧 Quick Production Frontend Validator
# Checks if backtesting controls are present and functional in the dashboard

import os
from pathlib import Path

def validate_frontend_controls():
    """Quick validation of frontend backtesting controls"""
    
    print("🔍 PRODUCTION FRONTEND VALIDATION")
    print("=" * 50)
    
    # Check which dashboard file is being served
    dashboard_file = Path("frontend/unified_dashboard.html")
    
    if not dashboard_file.exists():
        print("❌ CRITICAL: unified_dashboard.html not found!")
        print("   Expected at: frontend/unified_dashboard.html")
        return False
    
    print(f"✅ Dashboard file found: {dashboard_file}")
    print(f"   Size: {dashboard_file.stat().st_size:,} bytes")
    
    # Read and analyze the dashboard content
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for critical backtesting elements
    critical_elements = [
        'historical-data-manager',
        'availableDataSets',
        'backtestResultsList',
        'runHistoricalBacktest',
        'discoverAvailableData',
        'refreshHistoricalData'
    ]
    
    missing_elements = []
    for element in critical_elements:
        if element not in content:
            missing_elements.append(element)
    
    print(f"\n📋 BACKTESTING CONTROLS CHECK:")
    if missing_elements:
        print("❌ MISSING CRITICAL ELEMENTS:")
        for element in missing_elements:
            print(f"   - {element}")
        print("   This could explain why backtesting controls are gone!")
    else:
        print("✅ All critical backtesting elements found")
    
    # Check for API endpoints being called
    api_calls = [
        '/api/historical-data/discover',
        '/api/backtest/historical',
        '/api/backtest/history'
    ]
    
    print(f"\n🌐 API INTEGRATION CHECK:")
    missing_apis = []
    for api in api_calls:
        if api not in content:
            missing_apis.append(api)
        else:
            print(f"✅ {api}")
    
    if missing_apis:
        print("❌ MISSING API CALLS:")
        for api in missing_apis:
            print(f"   - {api}")
    
    # Check for JavaScript initialization
    js_functions = [
        'UnifiedDashboard',
        'initializeBacktestingUI',
        'fetchHistoricalBacktestStatus'
    ]
    
    print(f"\n⚙️ JAVASCRIPT CHECK:")
    missing_js = []
    for func in js_functions:
        if func not in content:
            missing_js.append(func)
        else:
            print(f"✅ {func}")
    
    if missing_js:
        print("❌ MISSING JS FUNCTIONS:")
        for func in missing_js:
            print(f"   - {func}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if missing_elements or missing_apis or missing_js:
        print("❌ FRONTEND ISSUES DETECTED")
        print("   Some backtesting controls may not be functioning properly")
        return False
    else:
        print("✅ FRONTEND APPEARS HEALTHY")
        print("   All critical components are present")
        return True

if __name__ == "__main__":
    validate_frontend_controls()