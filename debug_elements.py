#!/usr/bin/env python3
"""
🔍 Debug Dashboard Elements
Check if backtesting elements exist in the HTML and help identify the issue
"""

from pathlib import Path
import re

def find_elements():
    """Find backtesting-related elements in the HTML"""
    
    dashboard_path = Path('frontend/unified_dashboard.html')
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found")
        return
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for key elements
    elements_to_find = [
        'backtestPair',
        'backtestTimeframe', 
        'backtestDataStatus',
        'runBacktestBtn',
        'availableDataSets',
        'totalDataSets'
    ]
    
    print("🔍 DASHBOARD ELEMENT ANALYSIS")
    print("=" * 50)
    
    for element in elements_to_find:
        # Find all occurrences
        pattern = f'id="{element}"'
        matches = re.findall(pattern, content)
        
        # Also check for references
        ref_pattern = f'{element}'
        refs = len(re.findall(ref_pattern, content))
        
        if matches:
            print(f"✅ {element}: Found {len(matches)} definitions, {refs} total references")
        else:
            print(f"❌ {element}: NOT FOUND - this is the problem!")
    
    # Check for script sections that call these functions
    print(f"\n🔍 FUNCTION CALLS:")
    functions_to_find = [
        'discoverAvailableData',
        'updateBacktestingControls',
        'directDataDiscovery'
    ]
    
    for func in functions_to_find:
        refs = len(re.findall(func, content))
        print(f"📞 {func}: {refs} references")
    
    # Look for the backtesting section
    print(f"\n🔍 BACKTESTING SECTIONS:")
    backtest_sections = re.findall(r'<.*?backtest.*?>', content, re.IGNORECASE)
    for i, section in enumerate(backtest_sections[:5]):  # Show first 5
        print(f"📋 Section {i+1}: {section[:100]}...")

if __name__ == "__main__":
    find_elements()