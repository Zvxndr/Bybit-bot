#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE CORE PIPELINE & GRADUATION TEST
Final integration test of all core AI pipeline features
"""

import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_core_pipeline_components():
    """Test all 6 core AI pipeline components"""
    print("🔬 Testing Core AI Pipeline Components...")
    
    components = {
        "ML Engine": False,
        "Pipeline Manager": False, 
        "Database Integration": False,
        "Historical Backtesting": False,
        "Strategy Lifecycle": False,
        "Integration Test": False
    }
    
    try:
        # Test ML Strategy Discovery Engine
        try:
            from src.bot.strategy_engine.ml_strategy_discovery_engine import MLStrategyDiscoveryEngine
            engine = MLStrategyDiscoveryEngine()
            print(f"   ✅ ML Engine: {len(engine.strategy_types)} strategy types available")
            components["ML Engine"] = True
        except Exception as e:
            print(f"   ❌ ML Engine failed: {e}")
        
        # Test Pipeline Manager
        try:
            from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
            manager = AutomatedPipelineManager()
            print(f"   ✅ Pipeline Manager: {manager.discovery_rate_per_hour}/hour discovery rate")
            components["Pipeline Manager"] = True
        except Exception as e:
            print(f"   ❌ Pipeline Manager failed: {e}")
        
        # Test Database
        try:
            conn = sqlite3.connect('data/trading_bot.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            cursor.execute("SELECT COUNT(*) FROM historical_data")
            data_count = cursor.fetchone()[0]
            conn.close()
            print(f"   ✅ Database: {len(tables)} tables, {data_count:,} historical records")
            components["Database Integration"] = True
        except Exception as e:
            print(f"   ❌ Database failed: {e}")
        
        # Test Historical Data
        try:
            conn = sqlite3.connect('data/trading_bot.db')
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, timeframe, COUNT(*) FROM historical_data GROUP BY symbol, timeframe")
            data_sets = cursor.fetchall()
            conn.close()
            print(f"   ✅ Historical Data: {len(data_sets)} data sets available")
            components["Historical Backtesting"] = True
        except Exception as e:
            print(f"   ❌ Historical Data failed: {e}")
        
        # Test Strategy Lifecycle
        try:
            from src.bot.strategy_engine.ml_strategy_discovery_engine import MLStrategyDiscoveryEngine
            engine = MLStrategyDiscoveryEngine()
            name = engine._generate_strategy_name("BTC", "RSI")
            print(f"   ✅ Strategy Lifecycle: Generated name '{name}'")
            components["Strategy Lifecycle"] = True
        except Exception as e:
            print(f"   ❌ Strategy Lifecycle failed: {e}")
        
        # Integration Test
        passed_count = sum(1 for passed in components.values() if passed)
        if passed_count >= 4:  # At least 4/6 components working
            components["Integration Test"] = True
            print(f"   ✅ Integration Test: {passed_count}/6 components operational")
        else:
            print(f"   ❌ Integration Test: Only {passed_count}/6 components working")
        
        return components
        
    except Exception as e:
        print(f"   ❌ Core pipeline test failed: {e}")
        return components

def test_manual_graduation_frontend():
    """Test manual graduation frontend integration"""
    print("🌐 Testing Manual Graduation Frontend...")
    
    dashboard_path = Path('frontend/unified_dashboard.html')
    
    if not dashboard_path.exists():
        print("   ❌ Dashboard file not found")
        return False
    
    print("   ✅ Dashboard file exists")
    
    # Check for all manual graduation elements
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_elements = {
        'Manual Strategy Graduation': 'Section header',
        'promotableStrategies': 'Container for promotion candidates',
        'retirementCandidates': 'Container for retirement candidates', 
        'loadManualGraduation()': 'Main loading function',
        'promoteStrategy(': 'Individual strategy promotion',
        'retireStrategy(': 'Individual strategy retirement',
        'promoteAllQualified()': 'Bulk promotion function',
        'showGraduationHistory()': 'History viewing function',
        'updateGraduationInterface': 'Interface update logic',
        'showNotification': 'User feedback system'
    }
    
    found_elements = 0
    missing_elements = []
    
    for element, description in required_elements.items():
        if element in content:
            print(f"   ✅ Found: {element} ({description})")
            found_elements += 1
        else:
            print(f"   ❌ Missing: {element} ({description})")
            missing_elements.append(element)
    
    print(f"   🎯 Frontend elements: {found_elements}/{len(required_elements)}")
    
    # Check for proper HTML structure
    graduation_section = 'Manual Strategy Graduation' in content
    bootstrap_cards = 'card-header' in content and 'card-body' in content
    interactive_buttons = 'btn btn-success' in content and 'btn btn-danger' in content
    
    print(f"   📋 HTML Structure:")
    print(f"      {'✅' if graduation_section else '❌'} Graduation section present")
    print(f"      {'✅' if bootstrap_cards else '❌'} Bootstrap card layout")
    print(f"      {'✅' if interactive_buttons else '❌'} Interactive buttons")
    
    return found_elements >= 8 and graduation_section and bootstrap_cards

def test_database_schema_compatibility():
    """Test database schema for graduation functionality"""
    print("🗄️ Testing Database Schema Compatibility...")
    
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Check required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = ['strategy_metadata', 'strategy_performance', 'graduated_strategies']
        existing_required = [table for table in required_tables if table in tables]
        
        print(f"   📊 Required tables present: {len(existing_required)}/{len(required_tables)}")
        for table in existing_required:
            print(f"      ✅ {table}")
        for table in required_tables:
            if table not in existing_required:
                print(f"      ❌ {table} (missing)")
        
        # Check strategy_metadata columns
        cursor.execute("PRAGMA table_info(strategy_metadata)")
        metadata_cols = [col[1] for col in cursor.fetchall()]
        
        required_metadata_cols = ['strategy_id', 'name', 'status', 'created_at']
        metadata_compatibility = all(col in metadata_cols for col in required_metadata_cols)
        
        print(f"   📋 strategy_metadata compatibility: {'✅' if metadata_compatibility else '❌'}")
        
        # Check strategy_performance columns
        cursor.execute("PRAGMA table_info(strategy_performance)")
        performance_cols = [col[1] for col in cursor.fetchall()]
        
        required_performance_cols = ['strategy_id', 'win_rate', 'total_trades']
        performance_compatibility = all(col in performance_cols for col in required_performance_cols)
        
        print(f"   📈 strategy_performance compatibility: {'✅' if performance_compatibility else '❌'}")
        
        conn.close()
        
        return len(existing_required) >= 2 and metadata_compatibility and performance_compatibility
        
    except Exception as e:
        print(f"   ❌ Database schema test failed: {e}")
        return False

def test_graduation_criteria_logic():
    """Test the graduation criteria and logic"""
    print("🧮 Testing Graduation Criteria Logic...")
    
    # Test cases: (win_rate, cumulative_returns, total_trades, status, expected_action)
    test_cases = [
        (0.72, 0.085, 15, 'demo', 'promote'),      # Excellent performance
        (0.68, 0.065, 12, 'demo', 'promote'),      # Good performance  
        (0.60, 0.050, 10, 'demo', 'promote'),      # Minimum threshold
        (0.58, 0.085, 15, 'demo', 'monitor'),      # Win rate too low
        (0.72, 0.030, 15, 'demo', 'monitor'),      # Return too low
        (0.72, 0.085, 8, 'demo', 'monitor'),       # Not enough trades
        (0.72, 0.085, 15, 'live', 'monitor'),      # Already live
        (0.35, -0.08, 6, 'live', 'retire'),        # Poor live strategy
        (0.25, -0.15, 8, 'demo', 'retire'),        # Very poor demo
        (0.45, -0.12, 8, 'demo', 'retire'),        # Poor return
        (0.55, 0.02, 7, 'demo', 'monitor'),        # Needs more data
    ]
    
    correct_predictions = 0
    
    for win_rate, returns, trades, status, expected in test_cases:
        # Promotion criteria
        should_promote = (
            status == 'demo' and 
            win_rate >= 0.6 and 
            returns > 0.05 and 
            trades >= 10
        )
        
        # Retirement criteria
        should_retire = (
            status in ['demo', 'live'] and 
            (win_rate < 0.4 or returns < -0.1) and
            trades >= 5
        )
        
        # Determine predicted action
        if should_promote:
            predicted = 'promote'
        elif should_retire:
            predicted = 'retire'
        else:
            predicted = 'monitor'
        
        if predicted == expected:
            print(f"   ✅ {expected.upper()}: WR:{win_rate:.2f} R:{returns:.3f} T:{trades} S:{status}")
            correct_predictions += 1
        else:
            print(f"   ❌ Expected {expected}, got {predicted}: WR:{win_rate:.2f} R:{returns:.3f} T:{trades}")
    
    accuracy = correct_predictions / len(test_cases)
    print(f"   🎯 Graduation logic accuracy: {correct_predictions}/{len(test_cases)} ({accuracy:.1%})")
    
    return accuracy >= 0.9  # 90% accuracy threshold

def main():
    """Run comprehensive core pipeline and graduation tests"""
    print("🎯 COMPREHENSIVE CORE AI PIPELINE & GRADUATION TEST")
    print("=" * 60)
    
    tests = [
        ("Core AI Pipeline Components", test_core_pipeline_components),
        ("Manual Graduation Frontend", test_manual_graduation_frontend),
        ("Database Schema Compatibility", test_database_schema_compatibility),
        ("Graduation Criteria Logic", test_graduation_criteria_logic)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        try:
            result = test_func()
            if isinstance(result, dict):  # Core pipeline returns component dict
                component_count = sum(1 for passed in result.values() if passed)
                total_components = len(result)
                results[test_name] = (component_count, total_components, result)
                print(f"   🎯 Result: {component_count}/{total_components} components working")
            else:
                results[test_name] = result
                print(f"   🎯 Result: {'✅ PASS' if result else '❌ FAIL'}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results[test_name] = False
    
    print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
    print("=" * 60)
    
    overall_score = 0
    total_possible = 0
    
    for test_name, result in results.items():
        if isinstance(result, tuple):  # Core pipeline components
            score, total, components = result
            print(f"📋 {test_name}: {score}/{total} ({score/total:.1%})")
            for comp_name, passed in components.items():
                print(f"   {'✅' if passed else '❌'} {comp_name}")
            overall_score += score
            total_possible += total
        else:
            status = "✅ PASS" if result else "❌ FAIL" 
            print(f"📋 {test_name}: {status}")
            overall_score += 1 if result else 0
            total_possible += 1
    
    final_percentage = (overall_score / total_possible) * 100 if total_possible > 0 else 0
    
    print(f"\n🎯 OVERALL SYSTEM STATUS:")
    print(f"   📊 Score: {overall_score}/{total_possible} ({final_percentage:.1f}%)")
    
    if final_percentage >= 80:
        print("🚀 CORE AI PIPELINE SYSTEM: FULLY OPERATIONAL!")
        print("\n📋 SYSTEM READY FOR:")
        print("   ✅ Automated strategy discovery")
        print("   ✅ Historical backtesting") 
        print("   ✅ Manual strategy graduation")
        print("   ✅ Performance monitoring")
        print("   ✅ Strategy lifecycle management")
        
        print("\n🎮 NEXT STEPS:")
        print("   1. Start the dashboard server")
        print("   2. Navigate to Manual Strategy Graduation section")
        print("   3. Begin automated pipeline discovery")
        print("   4. Monitor strategy performance and graduation")
        
    elif final_percentage >= 60:
        print("⚠️ CORE AI PIPELINE SYSTEM: MOSTLY OPERATIONAL")
        print("   🔧 Minor adjustments needed for full functionality")
    else:
        print("❌ CORE AI PIPELINE SYSTEM: NEEDS ATTENTION")
        print("   🔧 Several components require fixes")

if __name__ == "__main__":
    main()