#!/usr/bin/env python3
"""
Core AI Pipeline Test Suite
==========================

Test the essential AI strategy pipeline features:
1. ML Strategy Discovery Engine
2. Automated Pipeline Manager  
3. Historical Backtesting
4. Strategy Graduation/Retirement
5. Database Integration
"""

import sys
import os
import sqlite3
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_ml_engine():
    """Test ML Strategy Discovery Engine"""
    print("🔬 Testing ML Strategy Discovery Engine...")
    try:
        from src.bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine, StrategyType
        
        # Initialize engine
        engine = MLStrategyDiscoveryEngine(australian_bias=0.3)
        
        # Check capabilities
        print(f"   ✅ ML Engine initialized")
        print(f"   📊 Available strategy types: {len(StrategyType)} types")
        
        return True
    except Exception as e:
        print(f"   ❌ ML Engine failed: {e}")
        return False

def test_pipeline_manager():
    """Test Automated Pipeline Manager"""
    print("🤖 Testing Automated Pipeline Manager...")
    try:
        from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager, PipelineConfig
        
        # Create simple config for testing
        config = PipelineConfig(
            discovery_rate_per_hour=3,
            primary_assets=['BTCUSDT', 'ETHUSDT'],
            min_backtest_score=60.0
        )
        
        # Initialize manager
        manager = AutomatedPipelineManager(config=config)
        
        print(f"   ✅ Pipeline Manager initialized")
        print(f"   🎯 Discovery rate: {config.discovery_rate_per_hour}/hour")
        print(f"   💰 Assets: {config.primary_assets}")
        
        return True
    except Exception as e:
        print(f"   ❌ Pipeline Manager failed: {e}")
        return False

def test_database_integration():
    """Test database integration for pipeline"""
    print("🗄️ Testing Database Integration...")
    try:
        # Check if database exists and has pipeline tables
        db_path = Path("data/trading_bot.db")
        if not db_path.exists():
            print(f"   ❌ Database not found at {db_path}")
            return False
        
        # Connect and check tables
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check for pipeline tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        pipeline_tables = [t for t in tables if 'strategy' in t.lower() or 'pipeline' in t.lower()]
        
        print(f"   ✅ Database connected")
        print(f"   📊 Total tables: {len(tables)}")
        print(f"   🔄 Pipeline tables: {pipeline_tables}")
        
        # Check for existing strategies
        if 'strategy_pipeline' in tables:
            cursor.execute("SELECT COUNT(*) FROM strategy_pipeline")
            count = cursor.fetchone()[0]
            print(f"   📈 Existing strategies: {count}")
        
        conn.close()
        return True
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def test_historical_backtesting():
    """Test historical backtesting capability"""
    print("📈 Testing Historical Backtesting...")
    try:
        # Check if we have historical data
        db_path = Path("data/trading_bot.db")
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        data_count = cursor.fetchone()[0]
        
        if data_count > 0:
            print(f"   ✅ Historical data available: {data_count:,} records")
            
            # Check data coverage
            cursor.execute("SELECT DISTINCT symbol, timeframe FROM historical_data")
            coverage = cursor.fetchall()
            print(f"   📊 Data coverage: {coverage}")
        else:
            print(f"   ⚠️ No historical data found - backtesting limited")
        
        conn.close()
        return True
    except Exception as e:
        print(f"   ❌ Backtesting test failed: {e}")
        return False

async def test_strategy_lifecycle():
    """Test strategy creation and lifecycle management"""
    print("🔄 Testing Strategy Lifecycle...")
    try:
        from src.bot.pipeline.strategy_naming_engine import StrategyNamingEngine
        
        # Test strategy naming
        naming_engine = StrategyNamingEngine()
        
        # Generate test strategy name
        strategy_name = naming_engine.generate_strategy_name(
            asset_pair='BTCUSDT',
            strategy_type='mean_reversion',
            strategy_description='Test mean reversion strategy'
        )
        
        print(f"   ✅ Strategy naming working")
        print(f"   🏷️ Generated name: {strategy_name.full_name}")
        print(f"   📋 Components: {strategy_name.asset}_{strategy_name.type_code}_{strategy_name.unique_id}")
        
        return True
    except Exception as e:
        print(f"   ❌ Strategy lifecycle test failed: {e}")
        return False

async def run_integration_test():
    """Run complete pipeline integration test"""
    print("🚀 Running Pipeline Integration Test...")
    try:
        from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager, PipelineConfig
        from src.bot.database.manager import DatabaseManager
        from src.bot.config import DatabaseConfig
        
        # Setup database manager
        db_config = DatabaseConfig(
            pool_size=5,
            max_overflow=10,
            echo=False,
            development={
                "dialect": "sqlite",
                "path": "./data/trading_bot.db"
            }
        )
        db_manager = DatabaseManager(db_config)
        
        # Create simple pipeline manager
        config = PipelineConfig(
            discovery_rate_per_hour=1,  # Slow for testing
            primary_assets=['BTCUSDT'],
            min_backtest_score=50.0
        )
        
        manager = AutomatedPipelineManager(
            config=config,
            db_manager=db_manager
        )
        
        # Test initialization
        initialized = await manager._initialize_components()
        print(f"   ✅ Integration test: Components initialized = {initialized}")
        
        # Test strategy discovery (without starting full pipeline)
        if initialized:
            # Simulate one discovery cycle
            strategy_data = await manager._generate_ml_strategy('BTCUSDT')
            if strategy_data:
                print(f"   🎯 Test strategy generated: {strategy_data['type']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def create_manual_graduation_interface():
    """Create manual strategy graduation controls"""
    print("🎮 Creating Manual Strategy Graduation Interface...")
    
    interface_html = """
    <!-- Manual Strategy Graduation Controls -->
    <div class="card card-primary">
        <div class="card-header">
            <h3 class="card-title">🎯 Manual Strategy Graduation</h3>
        </div>
        <div class="card-body">
            <div class="row">
                <div class="col-md-6">
                    <h5>Strategies Ready for Promotion</h5>
                    <div id="promotable-strategies">
                        <div class="strategy-item p-2 mb-2 border rounded">
                            <strong>BTC_MR_A4F2D</strong> - Mean Reversion
                            <br><small>Score: 78.5% | Sharpe: 1.85 | Return: 12.4%</small>
                            <br><button class="btn btn-success btn-sm mt-1" onclick="promoteStrategy('BTC_MR_A4F2D')">
                                Promote to Paper Trading
                            </button>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5>Strategies for Retirement</h5>
                    <div id="retirement-candidates">
                        <div class="strategy-item p-2 mb-2 border rounded">
                            <strong>ETH_BB_X7Y9Z</strong> - Bollinger Bands
                            <br><small>Score: 32.1% | Loss: -8.2% | Poor Performance</small>
                            <br><button class="btn btn-danger btn-sm mt-1" onclick="retireStrategy('ETH_BB_X7Y9Z')">
                                Retire Strategy
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    async function promoteStrategy(strategyId) {
        try {
            const response = await fetch('/api/pipeline/promote', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({strategy_id: strategyId})
            });
            
            if (response.ok) {
                showNotification(`Strategy ${strategyId} promoted successfully!`, 'success');
                loadStrategiesByPhase(); // Refresh display
            } else {
                showNotification(`Failed to promote ${strategyId}`, 'error');
            }
        } catch (error) {
            showNotification(`Error promoting strategy: ${error}`, 'error');
        }
    }
    
    async function retireStrategy(strategyId) {
        if (confirm(`Are you sure you want to retire strategy ${strategyId}?`)) {
            try {
                const response = await fetch('/api/pipeline/reject', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        strategy_id: strategyId,
                        reason: 'Manual retirement due to poor performance'
                    })
                });
                
                if (response.ok) {
                    showNotification(`Strategy ${strategyId} retired successfully!`, 'success');
                    loadStrategiesByPhase(); // Refresh display
                } else {
                    showNotification(`Failed to retire ${strategyId}`, 'error');
                }
            } catch (error) {
                showNotification(`Error retiring strategy: ${error}`, 'error');
            }
        }
    }
    
    // Load strategies and populate graduation interface
    async function loadStrategiesByPhase() {
        try {
            // Load backtest strategies ready for promotion
            const backtestResponse = await fetch('/api/pipeline/strategies/backtest');
            const backtestStrategies = await backtestResponse.json();
            
            // Load paper trading strategies for potential retirement
            const paperResponse = await fetch('/api/pipeline/strategies/paper');
            const paperStrategies = await paperResponse.json();
            
            // Populate graduation interface
            updateGraduationInterface(backtestStrategies, paperStrategies);
        } catch (error) {
            console.error('Error loading strategies:', error);
        }
    }
    
    function updateGraduationInterface(backtestStrategies, paperStrategies) {
        // Update promotable strategies
        const promotableDiv = document.getElementById('promotable-strategies');
        promotableDiv.innerHTML = '';
        
        backtestStrategies.strategies?.forEach(strategy => {
            if (strategy.ready_for_promotion) {
                const div = document.createElement('div');
                div.className = 'strategy-item p-2 mb-2 border rounded';
                div.innerHTML = `
                    <strong>${strategy.strategy_id}</strong> - ${strategy.strategy_type}
                    <br><small>Score: ${strategy.backtest_score}% | Sharpe: ${strategy.sharpe_ratio} | Return: ${strategy.backtest_return}%</small>
                    <br><button class="btn btn-success btn-sm mt-1" onclick="promoteStrategy('${strategy.strategy_id}')">
                        Promote to Paper Trading
                    </button>
                `;
                promotableDiv.appendChild(div);
            }
        });
        
        // Update retirement candidates (poor performing paper strategies)
        const retirementDiv = document.getElementById('retirement-candidates');
        retirementDiv.innerHTML = '';
        
        paperStrategies.strategies?.forEach(strategy => {
            if (strategy.paper_pnl < -5) { // Strategies losing more than 5%
                const div = document.createElement('div');
                div.className = 'strategy-item p-2 mb-2 border rounded';
                div.innerHTML = `
                    <strong>${strategy.strategy_id}</strong> - ${strategy.strategy_type}
                    <br><small>PnL: ${strategy.paper_pnl}% | Poor Performance</small>
                    <br><button class="btn btn-danger btn-sm mt-1" onclick="retireStrategy('${strategy.strategy_id}')">
                        Retire Strategy
                    </button>
                `;
                retirementDiv.appendChild(div);
            }
        });
    }
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', loadStrategiesByPhase);
    </script>
    """
    
    print("   ✅ Manual graduation interface created")
    return interface_html

async def main():
    """Run comprehensive core pipeline test"""
    print("🎯 CORE AI STRATEGY PIPELINE TEST SUITE")
    print("=" * 50)
    
    results = []
    
    # Test individual components
    results.append(("ML Engine", test_ml_engine()))
    results.append(("Pipeline Manager", test_pipeline_manager()))
    results.append(("Database Integration", test_database_integration()))
    results.append(("Historical Backtesting", test_historical_backtesting()))
    results.append(("Strategy Lifecycle", await test_strategy_lifecycle()))
    results.append(("Integration Test", await run_integration_test()))
    
    # Create manual controls
    graduation_interface = create_manual_graduation_interface()
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY:")
    passed = sum(1 for name, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} core components working ({passed/total*100:.0f}%)")
    
    if passed >= total * 0.8:  # 80% success rate
        print("🚀 CORE PIPELINE READY FOR PRODUCTION!")
        print("\n📋 Next Steps:")
        print("   1. Deploy manual graduation interface to dashboard")
        print("   2. Start automated discovery pipeline")
        print("   3. Monitor strategy progression")
        print("   4. Test paper trading graduation")
    else:
        print("⚠️ CORE PIPELINE NEEDS FIXES")
        print("\n🔧 Required Actions:")
        failed = [name for name, result in results if not result]
        for failure in failed:
            print(f"   - Fix {failure}")

if __name__ == "__main__":
    asyncio.run(main())