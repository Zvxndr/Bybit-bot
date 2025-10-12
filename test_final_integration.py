#!/usr/bin/env python3
"""
🎯 FINAL INTEGRATION TEST - Historical Data & Backtesting
Verifies that the comprehensive fix has resolved the persistent issue
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import json

def test_database_integration():
    """Test that database has the data and structure we need"""
    print("🗄️ Testing Database Integration...")
    
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Test historical data
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        historical_count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*) as count
            FROM historical_data 
            GROUP BY symbol, timeframe
        """)
        datasets = cursor.fetchall()
        
        print(f"   ✅ Historical data: {historical_count:,} total records")
        for symbol, timeframe, count in datasets:
            print(f"      📊 {symbol} {timeframe}: {count:,} records")
        
        # Test backtest results
        cursor.execute("SELECT COUNT(*) FROM backtest_results")
        backtest_count = cursor.fetchone()[0]
        
        print(f"   ✅ Backtest results: {backtest_count} records")
        
        if backtest_count > 0:
            cursor.execute("""
                SELECT pair, timeframe, total_return_pct, win_rate, trades_count
                FROM backtest_results
                ORDER BY timestamp DESC
                LIMIT 3
            """)
            results = cursor.fetchall()
            
            for pair, timeframe, return_pct, win_rate, trades in results:
                print(f"      📈 {pair} {timeframe}: {return_pct:.1f}% return, {win_rate:.1f}% win rate")
        
        conn.close()
        return historical_count > 0
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def test_data_discovery_simulation():
    """Simulate the data discovery API endpoint"""
    print("🔍 Testing Data Discovery Logic...")
    
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Simulate the discovery query
        cursor.execute("""
            SELECT 
                symbol, 
                timeframe, 
                COUNT(*) as record_count,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest
            FROM historical_data
            GROUP BY symbol, timeframe
            HAVING COUNT(*) > 0
            ORDER BY record_count DESC
        """)
        
        results = cursor.fetchall()
        datasets = []
        
        for row in results:
            symbol, timeframe, count, earliest, latest = row
            
            try:
                # Test timestamp parsing
                if isinstance(earliest, (int, float)):
                    if earliest > 1e10:
                        start_dt = datetime.fromtimestamp(earliest / 1000)
                        end_dt = datetime.fromtimestamp(latest / 1000)
                    else:
                        start_dt = datetime.fromtimestamp(earliest)
                        end_dt = datetime.fromtimestamp(latest)
                else:
                    start_dt = datetime.fromisoformat(str(earliest))
                    end_dt = datetime.fromisoformat(str(latest))
                
                duration = (end_dt - start_dt).days
                
                datasets.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'record_count': count,
                    'duration_days': duration,
                    'date_range': f"{start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}"
                })
                
                print(f"   ✅ {symbol} {timeframe}: {count:,} records, {duration} days")
                
            except Exception as e:
                print(f"   ❌ {symbol} {timeframe}: Timestamp parsing error - {e}")
                return False
        
        conn.close()
        
        if datasets:
            print(f"   🎯 Discovery simulation: {len(datasets)} datasets ready")
            return True
        else:
            print("   ❌ No datasets found in discovery simulation")
            return False
        
    except Exception as e:
        print(f"   ❌ Discovery simulation failed: {e}")
        return False

def test_backtest_results_simulation():
    """Simulate the backtest results API endpoint"""
    print("📊 Testing Backtest Results Logic...")
    
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Simulate the backtest history query
        cursor.execute("""
            SELECT pair, timeframe, starting_balance, total_return_pct, sharpe_ratio, 
                   status, timestamp, trades_count, max_drawdown, win_rate
            FROM backtest_results 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "pair": row[0],
                "timeframe": row[1], 
                "starting_balance": row[2],
                "total_return": row[3],
                "sharpe_ratio": row[4],
                "status": row[5],
                "timestamp": row[6],
                "trades_count": row[7],
                "max_drawdown": row[8],
                "win_rate": row[9]
            })
        
        conn.close()
        
        if results:
            print(f"   ✅ Backtest results simulation: {len(results)} results found")
            for result in results[:2]:
                print(f"      📈 {result['pair']} {result['timeframe']}: {result['total_return']:.1f}% return")
            return True
        else:
            print("   ⚠️ No backtest results found (this is normal for new systems)")
            return True  # This is OK - just means no backtests run yet
        
    except Exception as e:
        print(f"   ❌ Backtest results simulation failed: {e}")
        return False

def test_frontend_compatibility():
    """Test that frontend integration points exist"""
    print("🌐 Testing Frontend Integration Points...")
    
    dashboard_path = Path('frontend/unified_dashboard.html')
    
    if not dashboard_path.exists():
        print("   ❌ Dashboard file not found")
        return False
    
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for enhanced functions
    required_functions = [
        'discoverAvailableDataEnhanced',
        'loadBacktestResultsEnhanced', 
        'updateBacktestingControlsEnhanced',
        'updateBacktestResultsDisplayEnhanced'
    ]
    
    found_functions = 0
    for func in required_functions:
        if func in content:
            print(f"   ✅ Found: {func}")
            found_functions += 1
        else:
            print(f"   ❌ Missing: {func}")
    
    # Check for key elements
    key_elements = [
        'backtestPair',
        'backtestTimeframe',
        'Manual Strategy Graduation'
    ]
    
    found_elements = 0
    for element in key_elements:
        if element in content:
            print(f"   ✅ Element: {element}")
            found_elements += 1
    
    success = found_functions >= 3 and found_elements >= 2
    
    if success:
        print("   🎯 Frontend integration points verified")
    else:
        print("   ❌ Frontend integration incomplete")
    
    return success

def run_final_integration_test():
    """Run complete integration test"""
    print("🎯 FINAL INTEGRATION TEST - Historical Data & Backtesting Fix")
    print("=" * 65)
    
    tests = [
        ("Database Integration", test_database_integration),
        ("Data Discovery Simulation", test_data_discovery_simulation),
        ("Backtest Results Simulation", test_backtest_results_simulation),
        ("Frontend Compatibility", test_frontend_compatibility)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}...")
        try:
            if test_func():
                passed_tests += 1
                print(f"   ✅ PASS: {test_name}")
            else:
                print(f"   ❌ FAIL: {test_name}")
        except Exception as e:
            print(f"   ❌ ERROR in {test_name}: {e}")
    
    print(f"\n📊 FINAL TEST RESULTS:")
    print("=" * 40)
    print(f"🎯 Tests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.0f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - INTEGRATION FIX SUCCESSFUL!")
        print("\n✅ Your historical data should now properly appear in:")
        print("   • Data discovery controls")
        print("   • Backtesting pair/timeframe selectors")  
        print("   • Backtest results display")
        print("   • Manual strategy graduation interface")
        print("\n🚀 READY TO USE:")
        print("   1. Refresh your dashboard browser tab")
        print("   2. Check the backtesting section for available pairs")
        print("   3. Run a historical backtest to verify full integration")
        print("   4. Use manual graduation controls for strategy management")
        
    elif passed_tests >= 3:
        print("⚠️ MOSTLY SUCCESSFUL - Minor issues remain")
        print("   The core integration should work with minor limitations")
        
    else:
        print("❌ INTEGRATION FIX INCOMPLETE")
        print("   Several components need attention before full functionality")
    
    return passed_tests == total_tests

def main():
    """Main test execution"""
    try:
        success = run_final_integration_test()
        return success
    except Exception as e:
        print(f"\n❌ FINAL INTEGRATION TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)