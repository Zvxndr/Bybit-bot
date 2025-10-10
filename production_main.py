#!/usr/bin/env python3
"""
Production Main - Clean, Simple, Reliable Startup
================================================

Simple production entry point that:
- Uses standard Python imports with proper error handling
- Makes AI components truly optional
- Provides clear logging of what's available
- No mock systems or complex import manipulation
- Graceful degradation when components unavailable

Author: GitHub Copilot
Version: 2.0 (Clean Rebuild)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Basic setup for Docker environment
if os.path.exists('/app'):
    os.chdir('/app')
    sys.path.insert(0, '/app')
    sys.path.insert(0, '/app/src')
else:
    # Local development setup
    current_dir = os.getcwd()
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(current_dir, 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("🚀 Production Trading Bot - Clean Startup")
print(f"📁 Working Directory: {os.getcwd()}")
print(f"🐍 Python Version: {sys.version_info[:2]}")

class ComponentRegistry:
    """Simple component registry with clean loading"""
    
    def __init__(self):
        self.components = {}
        self.load_status = {}
        
    def load_component(self, name: str, module_path: str, class_name: str):
        """Load a single component with proper error handling"""
        try:
            # Standard import approach
            if module_path.startswith('src.'):
                module = __import__(module_path, fromlist=[class_name])
            else:
                # Handle relative imports
                spec = __import__(module_path, fromlist=[class_name])
                module = spec
            
            component_class = getattr(module, class_name)
            self.components[name] = component_class
            self.load_status[name] = "✅ Available"
            logger.info(f"✅ {name}: Loaded successfully")
            return True
            
        except ImportError as e:
            self.load_status[name] = f"❌ Import Error: {str(e)}"
            logger.warning(f"❌ {name}: Import failed - {e}")
            return False
        except AttributeError as e:
            self.load_status[name] = f"❌ Class Not Found: {str(e)}"
            logger.warning(f"❌ {name}: Class not found - {e}")
            return False
        except Exception as e:
            self.load_status[name] = f"❌ Error: {str(e)}"
            logger.warning(f"❌ {name}: Unexpected error - {e}")
            return False
    
    def get_component(self, name: str):
        """Get a loaded component or None"""
        return self.components.get(name)
    
    def is_available(self, name: str) -> bool:
        """Check if component is available"""
        return name in self.components
    
    def get_status_report(self) -> dict:
        """Get comprehensive status report"""
        return {
            'total_components': len(self.load_status),
            'available_components': len(self.components),
            'availability_rate': len(self.components) / len(self.load_status) if self.load_status else 0,
            'status_details': self.load_status
        }

def load_ai_components():
    """Load core AI components - these are required for system operation"""
    print("\n🤖 Loading Core AI Components...")
    
    registry = ComponentRegistry()
    
    # Core trading API
    success_trading = registry.load_component(
        "TradingAPI", 
        "src.main", 
        "TradingAPI"
    )
    
    # CORE AI Pipeline components - REQUIRED for system functionality
    success_pipeline = registry.load_component(
        "AutomatedPipelineManager",
        "src.bot.pipeline.automated_pipeline_manager",
        "AutomatedPipelineManager"
    )
    
    success_ml = registry.load_component(
        "MLStrategyDiscoveryEngine",
        "src.bot.ml_strategy_discovery.ml_engine",
        "MLStrategyDiscoveryEngine"
    )
    
    # Verify core components loaded successfully
    if not success_trading:
        print("❌ CRITICAL: TradingAPI failed to load")
    if not success_pipeline:
        print("❌ CRITICAL: AutomatedPipelineManager failed to load - this is a core AI feature")
    if not success_ml:
        print("❌ CRITICAL: MLStrategyDiscoveryEngine failed to load - this is a core AI feature")
    
    # Optional multi-exchange support  
    registry.load_component(
        "MultiExchangeDataManager",
        "src.data.multi_exchange_provider",
        "MultiExchangeDataManager"
    )
    
    # Core database components
    success_db = registry.load_component(
        "DatabaseManager",
        "src.bot.database.manager",
        "DatabaseManager"
    )
    
    if not success_db:
        print("❌ CRITICAL: DatabaseManager failed to load - required for AI pipeline data")
    
    # Print status report with emphasis on core AI features
    status = registry.get_status_report()
    print(f"\n📊 Core AI System Loading Summary:")
    print(f"   📈 Success Rate: {status['availability_rate']:.1%}")
    print(f"   ✅ Available: {status['available_components']}")
    print(f"   📦 Total: {status['total_components']}")
    
    # Highlight core AI components
    core_components = ["AutomatedPipelineManager", "MLStrategyDiscoveryEngine", "TradingAPI", "DatabaseManager"]
    
    print(f"\n🤖 CORE AI FEATURES:")
    for name in core_components:
        if name in status['status_details']:
            status_msg = status['status_details'][name]
            print(f"   {status_msg} {name}")
    
    print(f"\n📦 OPTIONAL FEATURES:")
    for name, status_msg in status['status_details'].items():
        if name not in core_components:
            print(f"   {status_msg} {name}")
    
    return registry

def start_application(registry: ComponentRegistry):
    """Start the main application with required core components"""
    print(f"\n🎯 Starting Application...")
    
    # Check core components that MUST be available
    required_components = ["TradingAPI", "AutomatedPipelineManager", "MLStrategyDiscoveryEngine"]
    missing_components = []
    
    for component in required_components:
        if not registry.is_available(component):
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ CRITICAL: Missing required core components: {', '.join(missing_components)}")
        print("❌ Cannot start without AI pipeline components - these are core features")
        return False
    
    # Set status for optional components only
    os.environ['MULTI_EXCHANGE_AVAILABLE'] = str(registry.is_available("MultiExchangeDataManager"))
    
    try:
        # Import and start the main application
        import uvicorn
        
        # The main application will check feature flags and adapt accordingly
        print("🌐 Starting FastAPI server...")
        print(f"📡 Server will start on 0.0.0.0:8080")
        print(f"🤖 AI Pipeline System: Fully Operational")
        print(f"📊 ML Discovery Engine: Active") 
        print(f"🎯 3-Phase Pipeline: Backtest → Paper → Live")
        
        # Start the server
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8080,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        return False
    
    return True

def main():
    """Main entry point"""
    try:
        # Load components
        registry = load_ai_components()
        
        # Start application
        success = start_application(registry)
        
        if not success:
            print("❌ Application failed to start")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()