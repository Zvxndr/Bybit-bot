#!/usr/bin/env python3
"""
Production AI Component Import Fix
=================================

Direct fix for AI component imports in Docker production environment.
This replaces the complex import strategies with a simple, direct approach.
"""

import os
import sys
import importlib.util
from pathlib import Path

def load_ai_component_directly(module_name, class_name, file_path):
    """Load AI component directly from file path - Docker optimized"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        # Load module directly
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            print(f"❌ Could not create spec for {module_name}")
            return None
            
        module = importlib.util.module_from_spec(spec)
        
        # Inject into sys.modules to handle circular imports
        sys.modules[module_name] = module
        sys.modules[f"src.{module_name.replace('.', '.')}"] = module
        
        # Execute the module
        spec.loader.exec_module(module)
        
        # Get the class
        if hasattr(module, class_name):
            component_class = getattr(module, class_name)
            print(f"✅ {class_name} loaded directly from {file_path}")
            return component_class
        else:
            print(f"❌ {class_name} not found in {module_name}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load {class_name}: {e}")
        return None

def load_all_ai_components():
    """Load all AI components using direct file loading"""
    print("🤖 LOADING AI COMPONENTS DIRECTLY")
    print("=" * 50)
    
    # Define AI components to load
    components = {
        'MultiExchangeDataManager': {
            'module_name': 'multi_exchange_provider',
            'class_name': 'MultiExchangeDataManager',
            'file_path': '/app/src/data/multi_exchange_provider.py'
        },
        'AutomatedPipelineManager': {
            'module_name': 'automated_pipeline_manager', 
            'class_name': 'AutomatedPipelineManager',
            'file_path': '/app/src/bot/pipeline/automated_pipeline_manager.py'
        },
        'MLStrategyDiscoveryEngine': {
            'module_name': 'ml_engine',
            'class_name': 'MLStrategyDiscoveryEngine', 
            'file_path': '/app/src/bot/ml_strategy_discovery/ml_engine.py'
        }
    }
    
    loaded_components = {}
    
    for component_name, config in components.items():
        print(f"\n📦 Loading {component_name}...")
        
        component_class = load_ai_component_directly(
            config['module_name'],
            config['class_name'], 
            config['file_path']
        )
        
        if component_class:
            loaded_components[component_name] = component_class
            print(f"   ✅ {component_name} ready for instantiation")
        else:
            loaded_components[component_name] = None
            print(f"   ❌ {component_name} failed to load")
    
    return loaded_components

def create_ai_component_instances(loaded_components):
    """Create instances of loaded AI components"""
    print(f"\n🏗️  CREATING AI COMPONENT INSTANCES")
    print("-" * 40)
    
    instances = {}
    
    # Create MultiExchangeDataManager instance
    if loaded_components.get('MultiExchangeDataManager'):
        try:
            multi_exchange_data = loaded_components['MultiExchangeDataManager']()
            instances['multi_exchange_data'] = multi_exchange_data
            print("✅ MultiExchangeDataManager instance created")
        except Exception as e:
            print(f"❌ MultiExchangeDataManager instantiation failed: {e}")
            instances['multi_exchange_data'] = None
    else:
        instances['multi_exchange_data'] = None
    
    # Create AutomatedPipelineManager (but don't instantiate until database ready)
    if loaded_components.get('AutomatedPipelineManager'):
        instances['AutomatedPipelineManager'] = loaded_components['AutomatedPipelineManager']
        print("✅ AutomatedPipelineManager class ready (instantiate after database)")
    else:
        instances['AutomatedPipelineManager'] = None
    
    # Create MLStrategyDiscoveryEngine (but don't instantiate until needed)
    if loaded_components.get('MLStrategyDiscoveryEngine'):
        instances['MLStrategyDiscoveryEngine'] = loaded_components['MLStrategyDiscoveryEngine']  
        print("✅ MLStrategyDiscoveryEngine class ready")
    else:
        instances['MLStrategyDiscoveryEngine'] = None
    
    return instances

if __name__ == "__main__":
    # Test the AI component loading
    os.chdir('/app')  # Ensure correct working directory
    
    print("🚀 AI COMPONENT DIRECT LOADING TEST")
    print("=" * 60)
    
    # Load components
    loaded_components = load_all_ai_components()
    
    # Create instances
    instances = create_ai_component_instances(loaded_components)
    
    # Report results
    print(f"\n📊 LOADING RESULTS:")
    for name, instance in instances.items():
        status = "✅ SUCCESS" if instance is not None else "❌ FAILED"
        print(f"   {name}: {status}")
    
    success_count = sum(1 for instance in instances.values() if instance is not None)
    total_count = len(instances)
    
    print(f"\n🎯 SUMMARY: {success_count}/{total_count} AI components loaded successfully")
    
    if success_count == total_count:
        print("🎉 ALL AI COMPONENTS LOADED - READY FOR INTEGRATION!")
    else:
        print("⚠️  Some AI components failed - check logs above for details")