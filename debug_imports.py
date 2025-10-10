#!/usr/bin/env python3
"""
Debug script to test AI component imports in isolation
"""

from __future__ import annotations
import sys
import os

# Setup paths
sys.path.insert(0, 'src')
os.environ['PYTHONPATH'] = 'src'

print("🔍 Testing AI Component Imports...")

# Test AutomatedPipelineManager
try:
    print("\n1️⃣ Testing AutomatedPipelineManager import...")
    from src.bot.pipeline.automated_pipeline_manager import AutomatedPipelineManager
    print("   ✅ AutomatedPipelineManager imported successfully!")
    
    # Test instantiation
    print("   🔧 Testing class instantiation...")
    pipeline = AutomatedPipelineManager()
    print("   ✅ AutomatedPipelineManager instantiated successfully!")
    
except Exception as e:
    import traceback
    print(f"   ❌ AutomatedPipelineManager failed: {e}")
    print(f"   📋 Full traceback:\n{traceback.format_exc()}")

# Test MLStrategyDiscoveryEngine
try:
    print("\n2️⃣ Testing MLStrategyDiscoveryEngine import...")
    from src.bot.ml_strategy_discovery.ml_engine import MLStrategyDiscoveryEngine
    print("   ✅ MLStrategyDiscoveryEngine imported successfully!")
    
    # Test instantiation
    print("   🔧 Testing class instantiation...")
    ml_engine = MLStrategyDiscoveryEngine()
    print("   ✅ MLStrategyDiscoveryEngine instantiated successfully!")
    
except Exception as e:
    import traceback
    print(f"   ❌ MLStrategyDiscoveryEngine failed: {e}")
    print(f"   📋 Full traceback:\n{traceback.format_exc()}")

print("\n🎯 Debug test complete!")