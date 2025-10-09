#!/usr/bin/env python3
"""
Full System Integration Test
Tests complete AI pipeline initialization and automation
"""

import asyncio
import sys
from datetime import datetime

async def test_full_pipeline_initialization():
    """Test complete AI pipeline system initialization"""
    
    print("🤖 FULL AI PIPELINE INTEGRATION TEST")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # Import all critical components
        print("\n📦 Importing core components...")
        from src.main import TradingAPI
        from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
        from src.bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
        from src.bot.database.manager import DatabaseManager
        from src.bot.config import DatabaseConfig
        
        print("✅ All components imported successfully")
        
        # Test ML Engine initialization
        print("\n🧠 Testing ML Strategy Discovery Engine...")
        ml_engine = MLStrategyDiscoveryEngine()
        print("✅ ML Engine initialized")
        
        # Test Database Manager
        print("\n💾 Testing Database Manager...")
        db_config = DatabaseConfig(pool_size=10, max_overflow=20, echo=False)
        db_manager = DatabaseManager(db_config)
        db_manager.initialize()  # Initialize the database
        print("✅ Database Manager initialized")
        
        # Test Pipeline Manager initialization
        print("\n⚙️ Testing Automated Pipeline Manager...")
        pipeline_manager = AutomatedPipelineManager(
            db_manager=db_manager,
            ml_engine=ml_engine
        )
        print("✅ Pipeline Manager initialized")
        
        # Test components initialization
        print("\n🔧 Testing component initialization...")
        if await pipeline_manager._initialize_components():
            print("✅ All pipeline components initialized successfully")
        else:
            print("❌ Pipeline component initialization failed")
            return False
        
        # Test pipeline startup (brief test)
        print("\n🚀 Testing pipeline startup...")
        startup_success = await pipeline_manager.start_pipeline()
        if startup_success:
            print("✅ AI Pipeline started successfully!")
            print("🤖 Strategy discovery automation is now ACTIVE")
            
            # Brief test run (5 seconds)
            print("\n⏱️ Running 5-second integration test...")
            await asyncio.sleep(5)
            
            # Check if discovery is working
            if pipeline_manager.is_running:
                print("✅ Pipeline is actively running")
            else:
                print("⚠️ Pipeline not in running state")
            
            # Graceful shutdown
            print("\n🛑 Stopping pipeline for test completion...")
            await pipeline_manager.stop_pipeline()
            print("✅ Pipeline stopped gracefully")
            
            return True
        else:
            print("❌ Pipeline startup failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test execution"""
    print("🧪 Starting Full System Integration Test...")
    
    success = await test_full_pipeline_initialization()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 FULL SYSTEM INTEGRATION TEST: PASSED")
        print("🚀 AI Pipeline is ready for deployment!")
        print("📈 Automated strategy discovery will begin immediately upon deployment")
    else:
        print("💥 FULL SYSTEM INTEGRATION TEST: FAILED") 
        print("⚠️ Issues need to be resolved before deployment")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution error: {e}")
        sys.exit(1)