#!/usr/bin/env python3
"""
Simplified Deployment Readiness Test
Focuses on critical import verification for deployment
"""

import sys
from datetime import datetime

def test_deployment_readiness():
    """Test deployment readiness with focus on Docker import compatibility"""
    
    print("🚀 DEPLOYMENT READINESS VERIFICATION")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {
        'critical_imports': [],
        'optional_imports': [],
        'failures': []
    }
    
    # Critical Component Tests (Must pass for deployment)
    critical_tests = [
        ('MultiExchangeDataManager', 'src.data.multi_exchange_provider', 'MultiExchangeDataManager'),
        ('AutomatedPipelineManager', 'src.bot.pipeline.automated_pipeline_manager', 'AutomatedPipelineManager'),
        ('MLStrategyDiscoveryEngine', 'src.bot.ml_strategy_discovery.ml_engine', 'MLStrategyDiscoveryEngine'),
        ('TradingAPI', 'src.main', 'TradingAPI'),
        ('DatabaseManager', 'src.bot.database.manager', 'DatabaseManager'),
        ('DatabaseModels', 'src.bot.database.models', 'StrategyPipeline')
    ]
    
    print("\n🔥 CRITICAL DEPLOYMENT IMPORTS")
    print("-" * 40)
    
    for name, module, component in critical_tests:
        try:
            exec(f"from {module} import {component}")
            print(f"✅ {name}: SUCCESS")
            results['critical_imports'].append(name)
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            results['failures'].append(f"{name}: {e}")
    
    # Optional Component Tests (Good to have but not blocking)
    optional_tests = [
        ('BybitEnhancedBacktestEngine', 'src.bot.backtesting.bybit_enhanced_backtest_engine', 'BybitEnhancedBacktestEngine'),
        ('BybitAPIClient', 'src.bybit_api', 'BybitAPIClient'),
        ('HistoricalDataManager', 'historical_data_downloader', 'HistoricalDataManager')
    ]
    
    print(f"\n⚡ OPTIONAL COMPONENTS")
    print("-" * 40)
    
    for name, module, component in optional_tests:
        try:
            exec(f"from {module} import {component}")
            print(f"✅ {name}: Available")
            results['optional_imports'].append(name)
        except Exception as e:
            print(f"⚠️ {name}: Not available - {e}")
    
    # AI Pipeline Compatibility Test
    print(f"\n🤖 AI PIPELINE COMPATIBILITY TEST")
    print("-" * 40)
    
    try:
        # Test if we can create the core AI components
        from src.bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
        from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
        from src.bot.database.manager import DatabaseManager
        from src.bot.config import DatabaseConfig
        
        # Test basic component creation (without initialization)
        ml_engine = MLStrategyDiscoveryEngine()
        db_config = DatabaseConfig(pool_size=5, max_overflow=10, echo=False)
        db_manager = DatabaseManager(db_config)
        pipeline_manager = AutomatedPipelineManager(
            db_manager=db_manager,
            ml_engine=ml_engine
        )
        
        print("✅ AI Pipeline Components: Compatible")
        print("✅ Core System: Ready for deployment")
        
    except Exception as e:
        print(f"❌ AI Pipeline: Compatibility issue - {e}")
        results['failures'].append(f"AI Pipeline: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ CRITICAL IMPORTS PASSED: {len(results['critical_imports'])}/6")
    for item in results['critical_imports']:
        print(f"   • {item}")
    
    print(f"\n⚡ OPTIONAL COMPONENTS: {len(results['optional_imports'])}/3")
    for item in results['optional_imports']:
        print(f"   • {item}")
    
    if results['failures']:
        print(f"\n❌ ISSUES FOUND: {len(results['failures'])}")
        for item in results['failures']:
            print(f"   • {item}")
    
    # Final verdict
    critical_passed = len(results['critical_imports']) == 6
    no_critical_failures = not any('MultiExchange' in f or 'Pipeline' in f or 'MLStrategy' in f for f in results['failures'])
    
    if critical_passed and no_critical_failures:
        print(f"\n🎉 DEPLOYMENT STATUS: READY ✅")
        print("🚀 All critical imports successful - Docker deployment should work!")
        print("📈 AI Pipeline automation will activate upon deployment")
        return True
    else:
        print(f"\n💥 DEPLOYMENT STATUS: NOT READY ❌")
        print("⚠️ Critical import issues must be resolved before deployment")
        return False

if __name__ == "__main__":
    success = test_deployment_readiness()
    sys.exit(0 if success else 1)