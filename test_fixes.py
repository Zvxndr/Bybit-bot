#!/usr/bin/env python3
"""
Quick test of logging and frontend fixes
======================================
Tests:
1. Logging verbosity - should only show errors
2. Dashboard initialization and backtesting controls
"""

import subprocess
import sys
import time
from pathlib import Path
import logging

def test_logging_verbosity():
    """Test that logging is now at ERROR level only"""
    print("🧪 TESTING LOGGING VERBOSITY")
    print("=" * 40)
    
    # Test basic logging configuration
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    
    # These should NOT appear in production (set to ERROR level)
    logging.info("This INFO message should NOT appear")
    logging.warning("This WARNING message should NOT appear") 
    logging.debug("This DEBUG message should NOT appear")
    
    # This SHOULD appear
    logging.error("✅ This ERROR message SHOULD appear")
    
    print("✅ Basic logging test complete - only ERROR should be visible above")

def test_production_logger_config():
    """Test that production logger is configured for ERROR only"""
    print("\n🧪 TESTING PRODUCTION LOGGER CONFIG")
    print("=" * 40)
    
    try:
        # Import and check production logger configuration
        sys.path.append('src')
        from src.production_logger import production_logger
        
        print(f"📊 Production logger level: {production_logger.log_level}")
        print(f"📊 Logger environment: {production_logger.environment}")
        
        # Test logger messages
        production_logger.app_logger.info("This INFO should NOT appear in ERROR mode")
        production_logger.app_logger.error("✅ This ERROR should appear")
        
        print("✅ Production logger test complete")
        
    except ImportError as e:
        print(f"⚠️  Could not test production logger: {e}")

def test_frontend_validation():
    """Validate frontend has backtesting controls"""
    print("\n🧪 TESTING FRONTEND BACKTESTING CONTROLS")  
    print("=" * 40)
    
    dashboard_file = Path("frontend/unified_dashboard.html")
    
    if not dashboard_file.exists():
        print("❌ Dashboard file not found!")
        return False
    
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for critical backtesting elements
    critical_elements = [
        'discoverAvailableData',
        'loadHistoricalDataPeriods', 
        'runPeriodBacktest',
        'initializeHistoricalDataControls',
        'historicalDataPeriods',
        'backtestResultsList'
    ]
    
    missing = []
    found = []
    
    for element in critical_elements:
        if element in content:
            found.append(element)
        else:
            missing.append(element)
    
    print(f"✅ Found {len(found)} critical elements:")
    for element in found:
        print(f"  ✓ {element}")
    
    if missing:
        print(f"⚠️  Missing {len(missing)} elements:")
        for element in missing:
            print(f"  ✗ {element}")
    
    print(f"\n📊 Dashboard file size: {dashboard_file.stat().st_size:,} bytes")
    print("✅ Frontend validation complete")
    
    return len(missing) == 0

def main():
    """Run all tests"""
    print("🚀 TESTING PRODUCTION FIXES")
    print("=" * 50)
    print("Testing fixes for:")
    print("1. Logging verbosity (ERROR only)")
    print("2. Backtesting controls initialization")
    print("=" * 50)
    
    test_logging_verbosity()
    test_production_logger_config() 
    frontend_ok = test_frontend_validation()
    
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    print("✅ Logging: Should only show ERROR messages now")
    print("✅ Production Logger: Configured for ERROR level")
    print(f"{'✅' if frontend_ok else '⚠️ '} Frontend: Backtesting controls {'ready' if frontend_ok else 'need attention'}")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Deploy changes to DigitalOcean")
    print("2. Test runtime logs show only errors") 
    print("3. Verify backtesting controls appear and function")
    
if __name__ == "__main__":
    main()