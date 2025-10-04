"""
Pipeline Integration Script

Integrates the AI Pipeline API with the existing trading bot system.
This script demonstrates how to add the pipeline endpoints to your FastAPI application.
"""

import asyncio
import logging
from fastapi import FastAPI
from ..bot.database.database_manager import DatabaseManager
from ..api.pipeline_api import PipelineAPI
from ..bot.pipeline import pipeline_manager


def integrate_pipeline_api(app: FastAPI, db_manager: DatabaseManager = None) -> PipelineAPI:
    """
    Integrate pipeline API endpoints into an existing FastAPI application.
    
    Args:
        app: FastAPI application instance
        db_manager: Database manager instance (optional)
        
    Returns:
        PipelineAPI instance for additional configuration
    """
    
    # Initialize database manager if not provided
    if db_manager is None:
        db_manager = DatabaseManager()
    
    # Create and register pipeline API
    pipeline_api = PipelineAPI(app, db_manager)
    
    logging.info("✅ AI Pipeline API integrated successfully")
    logging.info("📊 Available pipeline endpoints:")
    logging.info("   GET  /api/pipeline/status")
    logging.info("   GET  /api/pipeline/metrics") 
    logging.info("   GET  /api/pipeline/strategies/{phase}")
    logging.info("   POST /api/pipeline/control")
    logging.info("   POST /api/pipeline/strategy-action")
    logging.info("   GET  /api/pipeline/strategy/{strategy_id}/details")
    logging.info("   GET  /api/pipeline/analytics/overview")
    logging.info("   WS   /ws/pipeline")
    
    return pipeline_api


async def initialize_pipeline_system():
    """Initialize the complete AI pipeline system."""
    
    logging.info("🚀 Initializing AI Pipeline System...")
    
    try:
        # Start the pipeline manager
        success = await pipeline_manager.start_pipeline()
        
        if success:
            logging.info("✅ AI Pipeline System started successfully")
            logging.info("🔄 Pipeline is now discovering strategies automatically")
            
            # Get initial status
            status = await pipeline_manager.get_pipeline_status()
            logging.info(f"📊 Pipeline Status: {status['is_running']}")
            logging.info(f"🎯 Discovery Rate: {status['config']['discovery_rate_per_hour']}/hour")
            logging.info(f"📈 Active Strategies: {status['active_counts']}")
            
            return True
        else:
            logging.error("❌ Failed to start AI Pipeline System")
            return False
            
    except Exception as e:
        logging.error(f"❌ Pipeline initialization error: {e}")
        return False


async def demo_pipeline_integration():
    """Demonstration of complete pipeline integration."""
    
    print("🤖 AI Pipeline Integration Demo")
    print("=" * 50)
    
    # Create FastAPI app
    app = FastAPI(title="Trading Bot with AI Pipeline", version="1.0.0")
    
    # Add CORS middleware for development
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Integrate pipeline API
    pipeline_api = integrate_pipeline_api(app)
    
    # Initialize pipeline system
    success = await initialize_pipeline_system()
    
    if success:
        print("\n✅ Integration Complete!")
        print("\n🔗 Pipeline API Endpoints Available:")
        print("   📊 GET  /api/pipeline/status - Current pipeline status")
        print("   📈 GET  /api/pipeline/metrics - Real-time metrics")
        print("   🔍 GET  /api/pipeline/strategies/backtest - Backtest strategies")
        print("   📄 GET  /api/pipeline/strategies/paper - Paper trading strategies")
        print("   🚀 GET  /api/pipeline/strategies/live - Live trading strategies")
        print("   ⚡ POST /api/pipeline/control - Start/stop pipeline")
        print("   👆 POST /api/pipeline/strategy-action - Manual strategy actions")
        print("   🌐 WS   /ws/pipeline - Real-time updates")
        
        print("\n📡 WebSocket Events:")
        print("   🎯 strategy_discovered - New strategy found")
        print("   📈 strategy_promoted - Backtest → Paper")
        print("   🚀 strategy_graduated - Paper → Live")
        print("   ❌ strategy_rejected - Strategy removed")
        print("   📊 metrics_updated - Real-time metrics")
        
        print("\n🎮 Frontend Integration:")
        print("   The three-column pipeline UI is already connected")
        print("   WebSocket automatically updates strategy cards")
        print("   Real-time metrics update every 30 seconds")
        print("   Manual actions available via API calls")
        
        # Show current metrics
        metrics = pipeline_manager.get_current_metrics()
        print(f"\n📊 Current Pipeline Metrics:")
        print(f"   Strategies Tested Today: {metrics.strategies_tested_today}")
        print(f"   Candidates Found: {metrics.candidates_found_today}")
        print(f"   Success Rate: {metrics.success_rate_pct:.1f}%")
        print(f"   Backtest Queue: {metrics.backtest_count}")
        print(f"   Paper Trading: {metrics.paper_count}")
        print(f"   Live Trading: {metrics.live_count}")
        print(f"   Total Live P&L: ${metrics.total_live_pnl:.2f}")
        
        print(f"\n🔄 Pipeline Status: RUNNING")
        print(f"🎯 Discovering new USDT crypto strategies automatically")
        print(f"📈 Progressing strategies through validation pipeline")
        print(f"💰 Generating profits with validated strategies")
        
        return app, pipeline_api
    else:
        print("\n❌ Integration Failed!")
        return None, None


# Integration instructions for existing applications
def add_to_existing_app():
    """
    Instructions for adding pipeline to existing FastAPI app.
    
    Add this to your main FastAPI application file:
    
    ```python
    from src.bot.pipeline_integration import integrate_pipeline_api, initialize_pipeline_system
    
    # After creating your FastAPI app
    app = FastAPI()
    
    # Integrate pipeline API
    pipeline_api = integrate_pipeline_api(app)
    
    # Add to startup event
    @app.on_event("startup")
    async def startup_event():
        await initialize_pipeline_system()
        
    @app.on_event("shutdown") 
    async def shutdown_event():
        await pipeline_manager.stop_pipeline()
    ```
    
    The frontend is already configured to connect to these endpoints.
    The three-column pipeline UI will automatically start working.
    """
    pass


if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(demo_pipeline_integration())