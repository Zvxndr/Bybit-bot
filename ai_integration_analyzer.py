#!/usr/bin/env python3
"""
AI Component Integration Fix
===========================

Now that the main application is running, let's fix the AI component imports
without breaking the working deployment.
"""

import os
import sys
from pathlib import Path

def check_ai_component_files():
    """Check what AI component files actually exist in the container"""
    
    print("🔍 AI COMPONENT FILE ANALYSIS")
    print("=" * 50)
    
    # Check what's in the data directory
    data_dir = "/app/src/data"
    if os.path.exists(data_dir):
        print(f"📁 {data_dir} contents:")
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"   📄 {item} ({size} bytes)")
            else:
                print(f"   📁 {item}/")
    
    # Check critical AI files
    ai_files = [
        "/app/src/data/multi_exchange_provider.py",
        "/app/src/bot/pipeline/automated_pipeline_manager.py",
        "/app/src/bot/ml_strategy_discovery/ml_engine.py"
    ]
    
    print(f"\n🎯 CRITICAL AI FILES:")
    for file_path in ai_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size} bytes)")
            
            # Check if it has the expected classes
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if 'multi_exchange_provider' in file_path:
                    has_class = 'class MultiExchangeDataManager' in content
                    print(f"      MultiExchangeDataManager class: {'✅' if has_class else '❌'}")
                    
                elif 'automated_pipeline_manager' in file_path:
                    has_class = 'class AutomatedPipelineManager' in content  
                    print(f"      AutomatedPipelineManager class: {'✅' if has_class else '❌'}")
                    
                elif 'ml_engine' in file_path:
                    has_class = 'class MLStrategyDiscoveryEngine' in content
                    print(f"      MLStrategyDiscoveryEngine class: {'✅' if has_class else '❌'}")
                    
            except Exception as e:
                print(f"      ⚠️ Could not read file: {e}")
        else:
            print(f"   ❌ {file_path} - NOT FOUND")

def test_simple_imports():
    """Test if we can import the AI components now that the app is stable"""
    
    print(f"\n🧪 AI COMPONENT IMPORT TESTING")
    print("-" * 40)
    
    # Test MultiExchangeDataManager
    try:
        sys.path.insert(0, '/app/src')
        from data.multi_exchange_provider import MultiExchangeDataManager
        print("✅ MultiExchangeDataManager import successful!")
    except Exception as e:
        print(f"❌ MultiExchangeDataManager import failed: {e}")
    
    # Test AutomatedPipelineManager
    try:
        from bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
        print("✅ AutomatedPipelineManager import successful!")
    except Exception as e:
        print(f"❌ AutomatedPipelineManager import failed: {e}")
    
    # Test MLStrategyDiscoveryEngine  
    try:
        from bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
        print("✅ MLStrategyDiscoveryEngine import successful!")
    except Exception as e:
        print(f"❌ MLStrategyDiscoveryEngine import failed: {e}")

def create_ai_integration_patch():
    """Create a patch for main.py to better handle AI component imports"""
    
    print(f"\n🔧 CREATING AI INTEGRATION PATCH")
    print("-" * 40)
    
    patch_code = '''
# Enhanced AI Component Import Strategy
def load_ai_components_safely():
    """Load AI components with better error handling and fallbacks"""
    
    components = {}
    
    # Try MultiExchangeDataManager
    try:
        from data.multi_exchange_provider import MultiExchangeDataManager
        components['multi_exchange'] = MultiExchangeDataManager
        print("✅ MultiExchangeDataManager loaded successfully")
    except Exception as e:
        print(f"⚠️ MultiExchangeDataManager not available: {e}")
        # Create a minimal placeholder
        class MockMultiExchange:
            def __init__(self):
                self.name = "Mock Multi-Exchange Provider"
        components['multi_exchange'] = MockMultiExchange
    
    # Try AutomatedPipelineManager
    try:
        from bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
        components['pipeline_manager'] = AutomatedPipelineManager
        print("✅ AutomatedPipelineManager loaded successfully")
    except Exception as e:
        print(f"⚠️ AutomatedPipelineManager not available: {e}")
        # Create a minimal placeholder
        class MockPipelineManager:
            def __init__(self):
                self.name = "Mock Pipeline Manager"
        components['pipeline_manager'] = MockPipelineManager
    
    # Try MLStrategyDiscoveryEngine
    try:
        from bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
        components['ml_engine'] = MLStrategyDiscoveryEngine
        print("✅ MLStrategyDiscoveryEngine loaded successfully")
    except Exception as e:
        print(f"⚠️ MLStrategyDiscoveryEngine not available: {e}")
        # Create a minimal placeholder
        class MockMLEngine:
            def __init__(self):
                self.name = "Mock ML Strategy Engine"
        components['ml_engine'] = MockMLEngine
    
    return components

# Usage in main.py startup:
# ai_components = load_ai_components_safely()
'''
    
    try:
        with open('/tmp/ai_integration_patch.py', 'w') as f:
            f.write(patch_code)
        print("✅ AI integration patch created at /tmp/ai_integration_patch.py")
        print("   This can be integrated into main.py for better AI component handling")
    except Exception as e:
        print(f"❌ Could not create patch: {e}")

if __name__ == "__main__":
    os.chdir('/app')  # Ensure we're in the right directory
    
    print("🤖 AI COMPONENT INTEGRATION ANALYSIS")
    print("=" * 60)
    print("The main application is now running successfully!")
    print("Let's analyze and fix the AI component integration.\n")
    
    check_ai_component_files()
    test_simple_imports()
    create_ai_integration_patch()
    
    print(f"\n📋 SUMMARY:")
    print("✅ Core application: RUNNING")
    print("⚠️ AI components: Partial - needs import fixes")
    print("🎯 Next: Integrate AI components without breaking working app")