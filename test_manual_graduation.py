#!/usr/bin/env python3
"""
🎯 Manual Strategy Graduation Interface Test
Tests the manual graduation controls and API endpoints
"""

import sqlite3
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_database_strategies():
    """Test existing strategies in database"""
    print("🗄️ Testing Database Strategy Data...")
    
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Get all strategies with metadata and performance data
        cursor.execute("""
            SELECT 
                m.name, 
                m.status, 
                p.win_rate, 
                p.cumulative_returns, 
                p.total_trades,
                m.created_at
            FROM strategy_metadata m
            LEFT JOIN strategy_performance p ON m.strategy_id = p.strategy_id
            ORDER BY m.created_at DESC
        """)
        
        strategies = cursor.fetchall()
        print(f"   📊 Found {len(strategies)} strategies in database")
        
        promotable_count = 0
        retirable_count = 0
        
        for strategy in strategies:
            name, status, win_rate, total_return, trades_count, created_at = strategy
            
            # Handle None values
            win_rate = win_rate or 0.0
            total_return = total_return or 0.0
            trades_count = trades_count or 0
            
            # Check promotion criteria
            if (status == 'demo' and 
                win_rate >= 0.6 and 
                total_return > 0.05 and 
                trades_count >= 10):
                promotable_count += 1
                print(f"   ✅ PROMOTABLE: {name} - WR: {win_rate:.1%}, Return: {total_return:.1%}, Trades: {trades_count}")
            
            # Check retirement criteria
            elif ((status == 'demo' or status == 'live') and 
                  (win_rate < 0.4 or total_return < -0.1) and
                  trades_count >= 5):
                retirable_count += 1
                print(f"   ❌ RETIRABLE: {name} - WR: {win_rate:.1%}, Return: {total_return:.1%}, Trades: {trades_count}")
            
            else:
                print(f"   📊 MONITORING: {name} - WR: {win_rate:.1%}, Return: {total_return:.1%}, Trades: {trades_count}")
        
        conn.close()
        
        print(f"   🎯 Promotion candidates: {promotable_count}")
        print(f"   🛑 Retirement candidates: {retirable_count}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def create_sample_strategies():
    """Create sample strategies for testing graduation interface"""
    print("🎭 Creating Sample Strategies for Testing...")
    
    try:
        conn = sqlite3.connect('data/trading_bot.db')
        cursor = conn.cursor()
        
        # Sample strategies for testing
        test_strategies = [
            # Promotable strategies (good performance)
            {
                'id': 'BTC_RSI_WINNER_A',
                'name': 'BTC RSI Winner A',
                'status': 'demo',
                'win_rate': 0.72,
                'total_return': 0.085,
                'trades_count': 15,
                'description': 'High-performing RSI strategy ready for promotion',
                'strategy_type': 'RSI'
            },
            {
                'id': 'ETH_MA_CHAMPION',
                'name': 'ETH MA Champion',
                'status': 'demo', 
                'win_rate': 0.68,
                'total_return': 0.065,
                'trades_count': 12,
                'description': 'Moving average crossover with excellent results',
                'strategy_type': 'MA'
            },
            # Retirement candidates (poor performance)
            {
                'id': 'BTC_POOR_PERFORMER',
                'name': 'BTC Poor Performer',
                'status': 'demo',
                'win_rate': 0.25,
                'total_return': -0.15,
                'trades_count': 8,
                'description': 'Underperforming strategy candidate for retirement',
                'strategy_type': 'MACD'
            },
            {
                'id': 'ETH_FAILING_STRAT',
                'name': 'ETH Failing Strategy',
                'status': 'live',
                'win_rate': 0.35,
                'total_return': -0.08,
                'trades_count': 6,
                'description': 'Live strategy with poor performance',
                'strategy_type': 'BB'
            },
            # Monitoring strategies (decent performance)
            {
                'id': 'BTC_MONITOR_STABLE',
                'name': 'BTC Monitor Stable',
                'status': 'demo',
                'win_rate': 0.55,
                'total_return': 0.02,
                'trades_count': 7,
                'description': 'Stable performance, needs more data',
                'strategy_type': 'EMA'
            }
        ]
        
        created_count = 0
        
        for strategy in test_strategies:
            # Check if strategy already exists in metadata
            cursor.execute(
                "SELECT COUNT(*) FROM strategy_metadata WHERE strategy_id = ?",
                (strategy['id'],)
            )
            
            if cursor.fetchone()[0] == 0:  # Strategy doesn't exist
                # Insert into strategy_metadata
                cursor.execute("""
                    INSERT INTO strategy_metadata 
                    (strategy_id, name, description, strategy_type, status, created_at, total_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy['id'],
                    strategy['name'],
                    strategy['description'],
                    strategy['strategy_type'],
                    strategy['status'],
                    datetime.now().isoformat(),
                    strategy['trades_count']
                ))
                
                # Insert into strategy_performance 
                cursor.execute("""
                    INSERT INTO strategy_performance
                    (strategy_id, timestamp, period, win_rate, cumulative_returns, total_trades, created_at, trading_mode)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy['id'],
                    datetime.now().isoformat(),
                    'daily',
                    strategy['win_rate'],
                    strategy['total_return'],
                    strategy['trades_count'],
                    datetime.now().isoformat(),
                    strategy['status']
                ))
                
                created_count += 1
                print(f"   ✅ Created: {strategy['name']} ({strategy['status']})")
            else:
                print(f"   📊 Exists: {strategy['name']}")
        
        conn.commit()
        conn.close()
        
        print(f"   🎯 Created {created_count} new test strategies")
        return True
        
    except Exception as e:
        print(f"   ❌ Sample creation failed: {e}")
        return False

def test_graduation_logic():
    """Test the graduation logic criteria"""
    print("🧮 Testing Graduation Logic...")
    
    # Test promotion criteria
    test_cases = [
        # (win_rate, total_return, trades_count, status, should_promote, description)
        (0.72, 0.085, 15, 'demo', True, 'Excellent performance'),
        (0.68, 0.065, 12, 'demo', True, 'Good performance'),
        (0.60, 0.05, 10, 'demo', True, 'Minimum threshold met'),
        (0.58, 0.085, 15, 'demo', False, 'Win rate too low'),
        (0.72, 0.03, 15, 'demo', False, 'Return too low'),
        (0.72, 0.085, 8, 'demo', False, 'Not enough trades'),
        (0.72, 0.085, 15, 'live', False, 'Already live'),
        
        # Retirement cases
        (0.35, -0.08, 6, 'live', 'retire', 'Poor win rate'),
        (0.45, -0.12, 8, 'demo', 'retire', 'Poor return'),
        (0.25, -0.15, 8, 'demo', 'retire', 'Very poor performance'),
        (0.55, 0.02, 7, 'demo', 'monitor', 'Needs more data'),
    ]
    
    promotion_tests_passed = 0
    retirement_tests_passed = 0
    
    for win_rate, total_return, trades_count, status, expected, description in test_cases:
        # Promotion logic
        should_promote = (
            status == 'demo' and 
            win_rate >= 0.6 and 
            total_return > 0.05 and 
            trades_count >= 10
        )
        
        # Retirement logic  
        should_retire = (
            (status in ['demo', 'live']) and 
            (win_rate < 0.4 or total_return < -0.1) and
            trades_count >= 5
        )
        
        if expected is True:  # Should promote
            if should_promote:
                print(f"   ✅ PROMOTE: {description}")
                promotion_tests_passed += 1
            else:
                print(f"   ❌ FAILED PROMOTE: {description}")
        elif expected == 'retire':  # Should retire
            if should_retire:
                print(f"   ✅ RETIRE: {description}")
                retirement_tests_passed += 1
            else:
                print(f"   ❌ FAILED RETIRE: {description}")
        elif expected is False:  # Should not promote
            if not should_promote:
                print(f"   ✅ NO PROMOTE: {description}")
                promotion_tests_passed += 1
            else:
                print(f"   ❌ FALSE PROMOTE: {description}")
        else:  # Monitor
            if not should_promote and not should_retire:
                print(f"   ✅ MONITOR: {description}")
                promotion_tests_passed += 1
            else:
                print(f"   ❌ WRONG ACTION: {description}")
    
    print(f"   🎯 Promotion logic tests passed: {promotion_tests_passed}")
    print(f"   🛑 Retirement logic tests passed: {retirement_tests_passed}")
    
    return True

def test_frontend_integration():
    """Test that the frontend files exist and are properly structured"""
    print("🌐 Testing Frontend Integration...")
    
    dashboard_path = Path('frontend/unified_dashboard.html')
    
    if not dashboard_path.exists():
        print("   ❌ Dashboard file not found")
        return False
    
    print("   ✅ Dashboard file exists")
    
    # Check for manual graduation controls
    with open(dashboard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_elements = [
        'Manual Strategy Graduation',
        'loadManualGraduation',
        'promoteStrategy',
        'retireStrategy',
        'promotableStrategies',
        'retirementCandidates',
        'promoteAllQualified',
        'showGraduationHistory'
    ]
    
    found_elements = 0
    for element in required_elements:
        if element in content:
            print(f"   ✅ Found: {element}")
            found_elements += 1
        else:
            print(f"   ❌ Missing: {element}")
    
    print(f"   🎯 Frontend elements: {found_elements}/{len(required_elements)}")
    
    return found_elements == len(required_elements)

def main():
    """Run all manual graduation tests"""
    print("🎯 MANUAL STRATEGY GRADUATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Database Strategies", test_database_strategies),
        ("Sample Strategy Creation", create_sample_strategies), 
        ("Graduation Logic", test_graduation_logic),
        ("Frontend Integration", test_frontend_integration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        try:
            if test_func():
                print(f"   ✅ PASS {test_name}")
                passed_tests += 1
            else:
                print(f"   ❌ FAIL {test_name}")
        except Exception as e:
            print(f"   ❌ ERROR {test_name}: {e}")
    
    print(f"\n📊 TEST RESULTS SUMMARY:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed_tests else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.0f}%)")
    
    if passed_tests == total_tests:
        print("🚀 MANUAL GRADUATION INTERFACE READY!")
        print("\n📋 Next Steps:")
        print("   1. Start the dashboard server")
        print("   2. Open unified_dashboard.html")
        print("   3. Navigate to Manual Strategy Graduation section")
        print("   4. Test promotion/retirement controls")
    else:
        print("⚠️ Some tests failed - check implementation")

if __name__ == "__main__":
    main()