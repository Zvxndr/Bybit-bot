#!/usr/bin/env python3
"""
Deployment Verification Suite
Tests critical import functionality for Docker deployment
"""

import sys
import os
import traceback
from datetime import datetime

def test_critical_imports():
    """Test the imports that were failing in Docker"""
    
    print("🔧 DEPLOYMENT VERIFICATION SUITE")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    # Test 1: Core data manager import
    print("\n1️⃣ Testing MultiExchangeDataManager import...")
    try:
        from src.data.multi_exchange_provider import MultiExchangeDataManager
        print("✅ MultiExchangeDataManager imported successfully")
        results['passed'].append("MultiExchangeDataManager")
    except Exception as e:
        print(f"❌ MultiExchangeDataManager import failed: {e}")
        results['failed'].append(f"MultiExchangeDataManager: {e}")
    
    # Test 2: Pipeline manager import  
    print("\n2️⃣ Testing AutomatedPipelineManager import...")
    try:
        from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
        print("✅ AutomatedPipelineManager imported successfully")
        results['passed'].append("AutomatedPipelineManager")
    except Exception as e:
        print(f"❌ AutomatedPipelineManager import failed: {e}")
        results['failed'].append(f"AutomatedPipelineManager: {e}")
    
    # Test 3: ML Strategy Discovery Engine
    print("\n3️⃣ Testing MLStrategyDiscoveryEngine import...")
    try:
        from src.bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
        print("✅ MLStrategyDiscoveryEngine imported successfully")
        results['passed'].append("MLStrategyDiscoveryEngine")
    except Exception as e:
        print(f"❌ MLStrategyDiscoveryEngine import failed: {e}")
        results['failed'].append(f"MLStrategyDiscoveryEngine: {e}")
    
    # Test 4: Backtest engine
    print("\n4️⃣ Testing BybitEnhancedBacktestEngine import...")
    try:
        from src.bot.backtesting.bybit_enhanced_backtest_engine import BybitEnhancedBacktestEngine
        print("✅ BybitEnhancedBacktestEngine imported successfully")
        results['passed'].append("BybitEnhancedBacktestEngine")
    except Exception as e:
        print(f"❌ BybitEnhancedBacktestEngine import failed: {e}")
        results['failed'].append(f"BybitEnhancedBacktestEngine: {e}")
    
    # Test 5: Database manager
    print("\n5️⃣ Testing DatabaseManager import...")
    try:
        from src.bot.database.manager import DatabaseManager
        print("✅ DatabaseManager imported successfully")
        results['passed'].append("DatabaseManager")
    except Exception as e:
        print(f"❌ DatabaseManager import failed: {e}")
        results['failed'].append(f"DatabaseManager: {e}")
    
    # Test 6: Main API initialization
    print("\n6️⃣ Testing main TradingAPI import...")
    try:
        from src.main import TradingAPI
        print("✅ TradingAPI imported successfully")
        results['passed'].append("TradingAPI")
    except Exception as e:
        print(f"❌ TradingAPI import failed: {e}")
        results['failed'].append(f"TradingAPI: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEPLOYMENT VERIFICATION RESULTS")
    print("=" * 60)
    
    print(f"\n✅ PASSED ({len(results['passed'])}):")
    for item in results['passed']:
        print(f"   • {item}")
    
    if results['failed']:
        print(f"\n❌ FAILED ({len(results['failed'])}):")
        for item in results['failed']:
            print(f"   • {item}")
    
    if results['warnings']:
        print(f"\n⚠️ WARNINGS ({len(results['warnings'])}):")
        for item in results['warnings']:
            print(f"   • {item}")
    
    # Overall status
    if not results['failed']:
        print(f"\n🎉 ALL CRITICAL IMPORTS SUCCESSFUL!")
        print("🚀 Ready for AI pipeline activation in deployment")
        return True
    else:
        print(f"\n💥 {len(results['failed'])} CRITICAL IMPORTS FAILED")
        print("⚠️ Deployment will have issues - imports need fixing")
        return False

if __name__ == "__main__":
    success = test_critical_imports()
    sys.exit(0 if success else 1)